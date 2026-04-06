import numpy as np
import torch
import torch.nn as nn


def get_replacable_ids(tokenizer, filter_word, device):
    filter_lower = filter_word.lower()
    repl_ids = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        if token is None:
            continue
        if token.startswith("▁"):
            clean = token[1:]
            if clean.isascii() and clean.isalpha() and filter_lower not in token.lower():
                repl_ids.append(i)
        elif token.startswith("Ġ"):
            clean = token[1:]
            if clean.isascii() and clean.isalpha() and filter_lower not in token.lower():
                repl_ids.append(i)
    return torch.tensor(repl_ids, device=device)


def detect_encode_skip(tokenizer):
    return len(tokenizer.encode(""))


def suffix_roundtrips(tokenizer, suffix_list, skip):
    decoded = tokenizer.decode(suffix_list)
    encoded = tokenizer.encode(decoded)[skip:]
    return suffix_list == encoded


def get_lm(model):
    if hasattr(model, 'language_model'):
        return model.language_model
    return model


def get_loss(lm, input_ids_batch, loss_slice, target_ids, device):
    input_ids_t = torch.tensor(input_ids_batch, device=device)
    attention_mask = torch.ones_like(input_ids_t).long()
    logits = lm(input_ids=input_ids_t, attention_mask=attention_mask).logits
    target = torch.tensor(target_ids, device=device).unsqueeze(0).repeat(logits.shape[0], 1)
    return nn.CrossEntropyLoss(reduction='none')(
        logits[:, loss_slice, :].transpose(1, 2), target).sum(dim=-1)


def cal_replacable_ids(lm, embed_layer, suffix_ids, prefix_ids, postfix_ids, target_ids,
                       repl_ids, device, top_k=128, select_b=16):
    embed_weights = embed_layer.weight
    one_hot = torch.zeros(suffix_ids.shape[0], embed_weights.shape[0],
                          device=device, dtype=embed_weights.dtype)
    one_hot.scatter_(1, suffix_ids.unsqueeze(1),
                     torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype))
    one_hot.requires_grad_()

    suffix_embeds = (one_hot @ embed_weights).unsqueeze(0)
    postfix_embeds = embed_layer(torch.tensor(postfix_ids, device=device, dtype=torch.long).unsqueeze(0))
    target_embeds = embed_layer(torch.tensor(target_ids, device=device, dtype=torch.long).unsqueeze(0))
    # Structure: [prefix] [SUFFIX] [postfix] [target]
    parts = [suffix_embeds, postfix_embeds, target_embeds]
    if prefix_ids:
        prefix_embeds = embed_layer(torch.tensor(prefix_ids, device=device, dtype=torch.long).unsqueeze(0))
        parts.insert(0, prefix_embeds)
    full_embeds = torch.cat(parts, dim=1)
    n_before_target = sum(p.shape[1] for p in parts[:-1])
    loss_slice = slice(n_before_target - 1, full_embeds.shape[1] - 1)

    logits = lm(inputs_embeds=full_embeds).logits
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :],
                                  torch.tensor(target_ids, device=device))
    loss.backward()

    grad = one_hot.grad.clone()
    one_hot.requires_grad_(False)
    grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-12)
    ref_grad = torch.full_like(grad, -np.inf)
    ref_grad[:, repl_ids] = -grad[:, repl_ids]
    topk_indices = ref_grad.topk(top_k, dim=1).indices
    return topk_indices[:, torch.randperm(top_k, device=device)[:select_b]]


def select_prompt(lm, tokenizer, suffix_ids, prefix_ids, postfix_ids, target_ids,
                  prompt_replace_lst, filter_word, previous_loss, device,
                  enc_skip, batch_size=64):
    suffix_list = suffix_ids.tolist()
    filter_lower = filter_word.lower()

    filtered_lst = []
    for i in range(len(suffix_list)):
        pos_candidates = []
        for repl_token in prompt_replace_lst[i].tolist():
            if repl_token == suffix_list[i]:
                continue
            cand = suffix_list.copy()
            cand[i] = repl_token
            decoded = tokenizer.decode(cand)
            if filter_lower in decoded.lower():
                continue
            if not suffix_roundtrips(tokenizer, cand, enc_skip):
                continue
            pos_candidates.append(repl_token)
        filtered_lst.append(pos_candidates)

    index_lst = [(i, tok) for i, sublst in enumerate(filtered_lst) for tok in sublst]
    if not index_lst:
        return suffix_ids, previous_loss

    # Structure: prefix + suffix + postfix + target
    loss_start = len(prefix_ids) + len(suffix_list) + len(postfix_ids) - 1
    loss_end = loss_start + len(target_ids)
    loss_slice = slice(loss_start, loss_end)

    all_input_ids = []
    for pos, repl_token in index_lst:
        cand = suffix_list.copy()
        cand[pos] = repl_token
        all_input_ids.append(prefix_ids + cand + postfix_ids + target_ids)

    losses = []
    for i in range(0, len(all_input_ids), batch_size):
        batch = all_input_ids[i:i + batch_size]
        losses.append(get_loss(lm, batch, loss_slice, target_ids, device))
    losses = torch.cat(losses)

    sorted_indices = losses.argsort()
    new_suffix = suffix_ids.clone()
    used_pos = set()
    new_loss = previous_loss

    for idx in sorted_indices:
        pos, repl_token = index_lst[idx.item()]
        if pos in used_pos:
            continue
        old_val = new_suffix[pos].item()
        new_suffix[pos] = repl_token

        if not suffix_roundtrips(tokenizer, new_suffix.tolist(), enc_skip):
            new_suffix[pos] = old_val
            continue

        full_ids = prefix_ids + new_suffix.tolist() + postfix_ids + target_ids
        cur_loss = get_loss(lm, [full_ids], loss_slice, target_ids, device).item()
        if cur_loss < new_loss:
            new_loss = cur_loss
            used_pos.add(pos)
        else:
            new_suffix[pos] = old_val
            break

    return new_suffix, new_loss


def generate_suffix(model, tokenizer, question, target, filter_word,
                    num_epoch=256, token_nums=32, seed=42, device=None):
    if device is None:
        device = next(model.parameters()).device

    lm = get_lm(model)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed)
    repl_ids = get_replacable_ids(tokenizer, filter_word, device)
    enc_skip = detect_encode_skip(tokenizer)
    print(f"  Encode skip: {enc_skip}, replaceable tokens: {len(repl_ids)}", flush=True)

    for _ in range(1000):
        suffix_ids = repl_ids[torch.randperm(len(repl_ids), device=device)[:token_nums]]
        decoded = tokenizer.decode(suffix_ids.tolist())
        if filter_word not in decoded.lower() and suffix_roundtrips(tokenizer, suffix_ids.tolist(), enc_skip):
            break

    # Structure: [BOS] [SUFFIX] [question] [target]
    prefix_ids = tokenizer.encode("", add_special_tokens=True)  # BOS only
    postfix_ids = tokenizer.encode(f" simply answer: {question}", add_special_tokens=False)
    target_ids = tokenizer.encode(f" {target}", add_special_tokens=False)

    embed_layer = lm.get_input_embeddings()

    previous_loss = float('inf')
    best_loss = float('inf')
    best_suffix = suffix_ids.clone()

    for it in range(num_epoch):
        prompt_replace_lst = cal_replacable_ids(
            lm, embed_layer, suffix_ids, prefix_ids, postfix_ids, target_ids, repl_ids, device)
        suffix_ids, loss = select_prompt(
            lm, tokenizer, suffix_ids, prefix_ids, postfix_ids, target_ids,
            prompt_replace_lst, filter_word, previous_loss, device, enc_skip)

        if loss < best_loss:
            best_loss = loss
            best_suffix = suffix_ids.clone()

        print(f"  epoch {it}/{num_epoch}  loss: {previous_loss:.4f} -> {loss:.4f}  best: {best_loss:.4f}", flush=True)
        previous_loss = loss

    return best_suffix.tolist()
