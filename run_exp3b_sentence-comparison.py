# ~~~~~~~~~~~~~~~~~~~ EXPERIMENT 3B: SENTENCE COMPARISON

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import io

import torch
import gc

# Before anything heavy loads
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    TASK = "sentence_comparison"
    
    # Parse command-line arguments.
    args = io.parse_args()
    
    # Set random seed.
    np.random.seed(args.seed)

    # Meta information.
    meta_data = {
        "model": args.model,
        "seed": args.seed,
        "task": TASK,
        "eval_type": args.eval_type,
        "option_order": args.option_order,
        "data_file": args.data_file,
        "timestamp": io.timestamp()
    }
    
    # Set up model and other model-related variables.
    model = io.initialize_model(args)
    kwargs = {}
    
    # Read corpus data.
    df = pd.read_csv(args.data_file)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN LOOP
    results = []
    
    with torch.no_grad():
        for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
            good_sentence = row.good_sentence
            bad_sentence = row.bad_sentence

            if args.eval_type == "direct":
                # Get full-sentence probabilities.
                logprob_of_good_sentence = model.get_full_sentence_logprob(
                    good_sentence
                )
                logprob_of_bad_sentence = model.get_full_sentence_logprob(
                    bad_sentence
                )

                res = {
                    "item_id": row.item_id,
                    "good_sentence": good_sentence,
                    "bad_sentence": bad_sentence,
                    "logprob_of_good_sentence": logprob_of_good_sentence,
                    "logprob_of_bad_sentence": logprob_of_bad_sentence
                }

            else:
                # Determine option order.
                if args.option_order == "goodFirst":
                    options = [good_sentence, bad_sentence]
                else:
                    options = [bad_sentence, good_sentence]

                # Construct continuations.
                good_continuation = "1" if args.option_order == "goodFirst" else "2"
                bad_continuation = "2" if args.option_order == "goodFirst" else "1"

                # Get logprobs from the model.
                good_prompt, logprob_of_good_continuation, logprobs_good = \
                    model.get_logprob_of_continuation(
                        "",  # no prefix
                        good_continuation,
                        task=TASK,
                        options=options,
                        return_dist=True,
                        **kwargs
                    )

                bad_prompt, logprob_of_bad_continuation, logprobs_bad = \
                    model.get_logprob_of_continuation(
                        "",  # no prefix
                        bad_continuation,
                        task=TASK,
                        options=options,
                        return_dist=True,
                        **kwargs
                    )

                res = {
                    "item_id": row.item_id,
                    "good_prompt": good_prompt,
                    "good_sentence": good_sentence,
                    "bad_sentence": bad_sentence,
                    "good_continuation": good_continuation,
                    "bad_continuation": bad_continuation,
                    "logprob_of_good_continuation": logprob_of_good_continuation,
                    "logprob_of_bad_continuation": logprob_of_bad_continuation
                }

                if args.model_type == "openai":
                    res["top_logprobs"] = logprobs_good  # correct var
                elif args.dist_folder is not None:
                    model.save_dist_as_numpy(
                        logprobs_good,
                        f"{args.dist_folder}/{row.item_id}.npy"
                    )

            results.append(res)

            # clear memory explicitly
            torch.cuda.empty_cache()
            gc.collect()

    # Combine meta information with model results into one dict.
    output = {
        "meta": meta_data,
        "results": results
    }

    io.dict2json(output, args.out_file)
