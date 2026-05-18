import torch


def _list_save_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1


def batch_filter(batch, sep_id, pad_id, hidden_state=None):
    input_ids, attention_mask, pos, triples, ent_maps, sent_mask, _ = batch

    batch_size = input_ids.shape[0]
    device = input_ids.device

    input_ids = input_ids.cpu()
    attention_mask = attention_mask.cpu()
    pos = pos.cpu()
    triples = triples.cpu()
    ent_maps = ent_maps.cpu()
    sent_mask = sent_mask.cpu()
    if hidden_state is not None:
        hidden_state = hidden_state.cpu()

    f_input_ids, f_attention_mask, f_pos, f_triples, f_ent_maps, f_hidden_state = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for b in range(batch_size):
        sent_start, sent_end = pos[b, 0], pos[b, 1]
        sep_token_idx = input_ids[b].tolist().index(sep_id)
        ids = [0] + list(range(sent_start, sent_end)) + [sep_token_idx]

        f_input_ids.append(input_ids[b, ids])
        f_attention_mask.append(attention_mask[b, ids])
        f_pos.append(pos[b] - sent_start + 1)

        for t in triples[b]:
            t[0] = _list_save_index(ids, t[0].item())
            t[1] = _list_save_index(ids, t[1].item())
            t[2] = _list_save_index(ids, t[2].item())
            t[3] = _list_save_index(ids, t[3].item())
        f_triples.append(triples[b])
        f_ent_maps.append(ent_maps[b, ids])

        if hidden_state is not None:
            f_hidden_state.append(hidden_state[b, ids])

    max_len = max(len(x) for x in f_input_ids)
    for b in range(batch_size):
        pad = torch.zeros(max_len - len(f_input_ids[b]), dtype=torch.long)
        if hidden_state is not None:
            state_pad = torch.zeros(
                max_len - len(f_input_ids[b]), hidden_state.shape[-1]
            )
            f_hidden_state[b] = torch.cat([f_hidden_state[b], state_pad], dim=0)
        f_input_ids[b] = torch.cat([f_input_ids[b], pad.fill_(pad_id)])
        f_attention_mask[b] = torch.cat([f_attention_mask[b], pad])
        f_ent_maps[b] = torch.cat([f_ent_maps[b], pad])

    f_input_ids = torch.stack(f_input_ids).to(device)
    f_attention_mask = torch.stack(f_attention_mask).to(device)
    f_pos = torch.stack(f_pos).to(device)
    f_triples = torch.stack(f_triples).to(device)
    f_ent_maps = torch.stack(f_ent_maps).to(device)

    assert f_input_ids.shape == f_attention_mask.shape

    if hidden_state is not None:
        f_hidden_state = torch.stack(f_hidden_state).to(device)
        return (
            f_input_ids,
            f_attention_mask,
            f_pos,
            f_triples,
            f_ent_maps,
            sent_mask,
        ), f_hidden_state
    return (f_input_ids, f_attention_mask, f_pos, f_triples, f_ent_maps, sent_mask)
