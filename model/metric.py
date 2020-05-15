import torch
import torch.nn.functional as F


def recall(actual, predicted, k):
    act_set = set([actual])
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def compute_score(solution, prediction):
    n = len(solution)
    scores_r_10 = []
    scores_r_50 = []
    for i in range(n):
        assert solution[i]["candidate"] == prediction[i]["candidate"]

        scores_r_10.append(recall(solution[i]["target"], prediction[i]["ranking"], 10))
        scores_r_50.append(recall(solution[i]["target"], prediction[i]["ranking"], 50))

    return sum(scores_r_10) / n, sum(scores_r_50) / n


def sharded_cross_view_inner_product(vid_embds, text_embds, text_weights, subspaces,
                                     l2renorm=False, tol=1e-5, dist=False, val=False):
    B = text_embds[subspaces[0]].size(0)

    if len(vid_embds[subspaces[0]]) > 2:
        for key, value in vid_embds.items():
            vid_embds[key] = value.squeeze()

    device = vid_embds[subspaces[0]].device
    num_caps = text_embds[subspaces[0]].size(1)

    vid_val_dim = len(vid_embds[subspaces[0]].size()) - 1
    sims = torch.zeros(B * num_caps, vid_embds[subspaces[0]].size(vid_val_dim - 1), device=device)
    text_weights = text_weights.view(B * num_caps, -1)

    if l2renorm:
        l2_mass_vid, l2_mass_text = 0, 0
        for idx, modality in enumerate(subspaces):
            vid_embd_ = vid_embds[modality]
            # assert len(vid_embd_.size()) == 2, "expected B x feat_dim format"
            # l2_mass_vid += vid_embd_.reshape(B, -1).pow(2).sum(1)
            l2_mass_vid += vid_embd_.pow(2).sum(vid_val_dim)
            text_embd_ = text_embds[modality]
            # assert len(text_embd_.size()) == 3, "expected B x caps x feat_dim format"
            text_embd_ = text_embd_.reshape(B * num_caps, -1)
            text_embd_ = text_weights[:, idx:idx + 1] * text_embd_
            l2_mass_text += text_embd_.pow(2).sum(1)
        l2_mass_vid = torch.sqrt(l2_mass_vid.clamp(min=1e-6)).unsqueeze(vid_val_dim)
        l2_mass_text = torch.sqrt(l2_mass_text.clamp(min=1e-6)).unsqueeze(1)
    else:
        l2_mass_text, l2_mass_vid = 1, 1

    for idx, modality in enumerate(subspaces):
        vid_embd_ = vid_embds[modality] / l2_mass_vid
        text_embd_ = text_embds[modality].view(B * num_caps, -1)
        text_embd_ = text_weights[:, idx: idx + 1] * text_embd_
        msg = "expected weights to be applied to text embeddings"
        assert text_embd_.shape[0] == text_weights.shape[0], msg
        text_embd_ = text_embd_ / l2_mass_text
        if val:
            sims += (vid_embd_ - text_embd_.unsqueeze(1)).pow(2).sum(2)
        else:
            sims += torch.matmul(text_embd_, vid_embd_.t())  # (B x num_caps) x (B)

    if torch.isnan(sims).sum().item():
        import ipdb; ipdb.set_trace()
        raise ValueError("Found nans in similarity matrix!")

    return sims
