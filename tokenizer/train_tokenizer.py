from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors 

def train_tokenizer(
        files: list[str], 
        vocab_size: int = 16000, 
        save_path: str = "tokenizer/domain_tokenizer.json",
        min_frequency: int = 2,
):  
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    trainer = trainers.BpeTrainer(
        vocab_size = vocab_size, 
        min_frequency = min_frequency, 
        special_tokens = special_tokens,
        show_progress = True, 
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )


    tokenizer.train(files, trainer)
    tokenizer.save(save_path)
    print(f"Saved tokenizer to {save_path} with vocab size {vocab_size} and min frequency {min_frequency}.")
    return tokenizer 



def load_tokenizer(path: str = "tokenizer/domain_tokenizer.json") -> Tokenizer:
    return Tokenizer.from_file(path)


def run_diagnostics(tokenizer: Tokenizer, sample_texts: list[str]):
    """
    The four checks from before, now as runnable code.
    Run this immediately after training, before you trust the tokenizer.
    """
 
    # compression ratio
    total_chars, total_tokens = 0, 0
    for text in sample_texts:
        enc = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(enc.ids)
    ratio = total_chars / total_tokens if total_tokens else 0

    print(f" Compression ratio: {ratio:.2f} chars/token")

    if ratio < 3:
        print("    -> WARNING: low ratio, vocab may be too fragmented for this domain")
    elif ratio > 6:
        print("    -> WARNING: unusually high ratio, check for repeated boilerplate")
    else:
        print("    -> looks healthy")
 
    # common word fragmentation
    print("\n Common word fragmentation:")
    for w in ["the", "and", "model", "language", "transformer"]:
        toks = tokenizer.encode(w).tokens
        print(f"    {w!r:15s} -> {toks}")
 
    # round-trip correctness, important one
    print("\n Round-trip correctness:")
    all_pass = True
    for text in sample_texts:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        ok = decoded == text
        all_pass = all_pass and ok
        if not ok:
            print(f"    MISMATCH:\n      original: {text!r}\n      decoded:  {decoded!r}")
    
    print(f"    All samples round-trip correctly: {all_pass}")
 
    print(f"\n Vocab size: {tokenizer.get_vocab_size()}")
 


if __name__ == "__main__":
    files = ["data/cleaned_corpus.txt"]

    tok = train_tokenizer(
        files=files,
        vocab_size=16000,  # or whatever you've settled on for your domain
        save_path="tokenizer/domain_tokenizer.json",
    )

    samples = [
        "your first real sample sentence here",
        "your second real sample sentence here",
    ]
    run_diagnostics(tok, samples)