"""``whisperlite`` command-line interface.

Subcommands cover the full model lifecycle:

* ``tokenizer train``   — learn a BPE vocabulary from manifest transcripts
* ``manifest validate`` — sanity-check a JSONL manifest before training
* ``train``             — run a training job from a YAML config
* ``eval``              — compute WER/CER for a checkpoint on a manifest
* ``transcribe``        — transcribe audio files locally
* ``serve``             — start the HTTP API server
* ``version``           — print the package version

All commands exit 0 on success, 2 on expected/domain errors (bad input,
missing files), and let unexpected errors surface with a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from whisperlite.version import __version__

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisperlite",
        description="Train, evaluate, and serve a Whisper-style ASR model.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the package version")

    # tokenizer train
    tokenizer_parser = subparsers.add_parser("tokenizer", help="tokenizer utilities")
    tokenizer_sub = tokenizer_parser.add_subparsers(dest="tokenizer_command", required=True)
    tok_train = tokenizer_sub.add_parser("train", help="train a byte-level BPE tokenizer")
    tok_train.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="JSONL manifest whose 'text' fields form the corpus (repeatable)",
    )
    tok_train.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="plain-text corpus file, one utterance per line (repeatable)",
    )
    tok_train.add_argument("--vocab-size", type=int, required=True, help="total vocabulary size")
    tok_train.add_argument("--min-frequency", type=int, default=2)
    tok_train.add_argument("--output", required=True, help="output tokenizer JSON path")

    # manifest validate
    manifest_parser = subparsers.add_parser("manifest", help="manifest utilities")
    manifest_sub = manifest_parser.add_subparsers(dest="manifest_command", required=True)
    man_validate = manifest_sub.add_parser("validate", help="validate a JSONL manifest")
    man_validate.add_argument("path", help="manifest path")
    man_validate.add_argument(
        "--check-audio", action="store_true", help="also verify every audio file exists"
    )

    # train
    train_parser = subparsers.add_parser("train", help="run a training job")
    train_parser.add_argument("--config", required=True, help="training YAML config")
    train_parser.add_argument("--resume", default=None, help="checkpoint to resume from")

    # eval
    eval_parser = subparsers.add_parser("eval", help="evaluate a checkpoint (WER/CER)")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--manifest", required=True)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--beam-size", type=int, default=1)
    eval_parser.add_argument(
        "--max-utterances", type=int, default=None, help="cap the number of utterances scored"
    )

    # transcribe
    transcribe_parser = subparsers.add_parser("transcribe", help="transcribe audio files")
    transcribe_parser.add_argument("--checkpoint", required=True)
    transcribe_parser.add_argument("audio", nargs="+", help="audio file(s) to transcribe")
    transcribe_parser.add_argument("--device", default="auto")
    transcribe_parser.add_argument("--beam-size", type=int, default=1)
    transcribe_parser.add_argument("--temperature", type=float, default=0.0)
    transcribe_parser.add_argument(
        "--output-json", default=None, help="write results to this JSON file instead of stdout"
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="start the HTTP API server")
    serve_parser.add_argument(
        "--checkpoint", default=None, help="checkpoint path (or WHISPERLITE_CHECKPOINT)"
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", default=None, help="override WHISPERLITE_DEVICE")

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from whisperlite.data.manifest import read_manifest
    from whisperlite.text.tokenizer import BPETokenizer

    if not args.manifest and not args.text_file:
        raise ValueError("provide at least one --manifest or --text-file")

    texts: list[str] = []
    for manifest_path in args.manifest:
        texts.extend(entry.text for entry in read_manifest(manifest_path))
    for text_path in args.text_file:
        path = Path(text_path)
        if not path.is_file():
            raise ValueError(f"text file not found: {path}")
        texts.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    logger.info("training tokenizer on %d lines (vocab_size=%d)", len(texts), args.vocab_size)
    tokenizer = BPETokenizer.train(
        texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tokenizer.save(args.output)
    logger.info("saved tokenizer with %d tokens to %s", tokenizer.vocab_size, args.output)
    return 0


def _cmd_manifest_validate(args: argparse.Namespace) -> int:
    from whisperlite.data.manifest import read_manifest

    entries = read_manifest(args.path)
    missing = 0
    if args.check_audio:
        for entry in entries:
            if not Path(entry.audio_filepath).is_file():
                logger.error("missing audio file: %s", entry.audio_filepath)
                missing += 1
    durations = [entry.duration for entry in entries if entry.duration is not None]
    total_hours = sum(durations) / 3600 if durations else None
    print(
        json.dumps(
            {
                "entries": len(entries),
                "with_duration": len(durations),
                "total_hours": round(total_hours, 2) if total_hours is not None else None,
                "missing_audio": missing if args.check_audio else None,
            }
        )
    )
    if missing:
        raise ValueError(f"{missing} audio files referenced by the manifest are missing")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import dataclasses

    from whisperlite.config import load_train_config
    from whisperlite.training.trainer import Trainer

    config = load_train_config(args.config)
    if args.resume:
        config = dataclasses.replace(config, resume_from=args.resume)
    summary = Trainer(config).train()
    logger.info("training complete: %s", summary)
    print(json.dumps(summary))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import torch

    from whisperlite.audio.features import load_audio, log_mel_spectrogram, pad_or_trim
    from whisperlite.data.manifest import read_manifest
    from whisperlite.model.checkpoint import load_model
    from whisperlite.model.generation import GenerationOptions, generate
    from whisperlite.training.metrics import char_error_rate, word_error_rate
    from whisperlite.utils import resolve_device

    device = resolve_device(args.device)
    model, tokenizer, audio_config = load_model(args.checkpoint, device)
    entries = read_manifest(args.manifest)
    if args.max_utterances is not None:
        entries = entries[: args.max_utterances]
    options = GenerationOptions(beam_size=args.beam_size)

    references: list[str] = []
    hypotheses: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(entries), args.batch_size):
            batch_entries = entries[start : start + args.batch_size]
            mels = torch.stack(
                [
                    log_mel_spectrogram(
                        pad_or_trim(
                            load_audio(entry.audio_filepath, audio_config.sample_rate),
                            audio_config.chunk_samples,
                        ),
                        audio_config,
                    )
                    for entry in batch_entries
                ]
            ).to(device)
            results = generate(model, tokenizer, mels, options)
            for entry, result in zip(batch_entries, results, strict=True):
                references.append(entry.text)
                hypotheses.append(result.text)
            logger.info("scored %d/%d utterances", len(references), len(entries))

    report = {
        "utterances": len(references),
        "wer": round(word_error_rate(references, hypotheses), 4),
        "cer": round(char_error_rate(references, hypotheses), 4),
    }
    print(json.dumps(report))
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from whisperlite.model.checkpoint import load_model
    from whisperlite.model.generation import GenerationOptions
    from whisperlite.transcribe import transcribe_file
    from whisperlite.utils import resolve_device

    device = resolve_device(args.device)
    model, tokenizer, audio_config = load_model(args.checkpoint, device)
    options = GenerationOptions(beam_size=args.beam_size, temperature=args.temperature)

    results = []
    for audio_path in args.audio:
        output = transcribe_file(model, tokenizer, audio_path, audio_config, options=options)
        results.append(
            {
                "audio_filepath": audio_path,
                "text": output.text,
                "duration": output.duration,
                "chunks": [chunk.__dict__ for chunk in output.chunks],
            }
        )
        logger.info("%s -> %r", audio_path, output.text)

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    from whisperlite.serving.app import create_app
    from whisperlite.serving.settings import ServingSettings

    env = dict(os.environ)
    if args.checkpoint:
        env["WHISPERLITE_CHECKPOINT"] = args.checkpoint
    if args.device:
        env["WHISPERLITE_DEVICE"] = args.device
    settings = ServingSettings.from_env(env)
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from whisperlite.logging_utils import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "version":
        print(__version__)
        return 0

    handlers = {
        "train": _cmd_train,
        "eval": _cmd_eval,
        "transcribe": _cmd_transcribe,
        "serve": _cmd_serve,
    }
    try:
        if args.command == "tokenizer":
            return _cmd_tokenizer_train(args)
        if args.command == "manifest":
            return _cmd_manifest_validate(args)
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        # Domain errors (ConfigError, ManifestError, TokenizerError, ...) all
        # derive from ValueError: report cleanly without a traceback.
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
