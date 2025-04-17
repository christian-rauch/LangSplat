#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import argparse
from natsort import natsorted
import open_clip

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--feature_path', type=str, default="language_features_decode")
    parser.add_argument("-q", "--query", type=str, required=True,
                        help="optional query with similarity visualisation")
    parser.add_argument("-e", "--export_path", type=str,)
    args = parser.parse_args()

    feature_dir = f"{args.dataset_path}/{args.feature_path}"

    # get embedding for text query
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="laion2b_s34b_b88k",
        precision="fp16",
    )
    clip_model.eval()
    clip_model.to("cuda:0")

    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    text = clip_tokenizer([args.query]).to(device="cuda:0")
    textfeat = clip_model.encode_text(text)
    textfeat = torch.nn.functional.normalize(textfeat, dim=-1)

    export_path_sim = os.path.join(args.export_path, "ls", args.query.replace(' ', '_'))
    os.makedirs(export_path_sim, exist_ok=True)

    # assemble full reslution embedding map from segments and their embedding
    segment_files = natsorted(glob.glob(os.path.join(feature_dir, "*_s.npy")))
    embedding_files = natsorted(glob.glob(os.path.join(feature_dir, "*_f.npy")))
    for ifile, (s, f) in enumerate(zip(segment_files, embedding_files, strict=True)):
        segm = torch.from_numpy(np.load(s)[3].astype(int)).to("cuda:0")
        invaid = segm < 0

        feat = torch.from_numpy(np.load(f)).to("cuda:0")

        embeddings = torch.zeros(segm.shape[0], segm.shape[1], feat.shape[-1], device="cuda:0")
        for i in range(torch.max(segm)):
            m = (segm == i)
            embeddings[m] = feat[i]

        embeddings[invaid] = torch.nan

        # cosine similarity to text query
        similarity = cosine_similarity(embeddings, textfeat)
        similarity[invaid] = torch.nan

        np.save(os.path.join(export_path_sim, f"similarity_{ifile}.npy"), similarity.detach().cpu().numpy())
