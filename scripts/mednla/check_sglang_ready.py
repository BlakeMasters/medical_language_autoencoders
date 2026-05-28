"""Check that a SGLang server can serve MedNLA AV decodes."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.decode import apply_sglang_auth_token, resolve_checkpoint_path
from nla_inference import NLAClient


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _resolve_model(cfg: dict[str, Any], short_name: str) -> dict[str, Any]:
    matches = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and model.get("short_name") == short_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one model with short_name={short_name!r}, got {len(matches)}")
    return matches[0]


def _resolve_auth_token(args: argparse.Namespace) -> str | None:
    if args.auth_token and args.auth_token_env:
        raise ValueError("--auth-token and --auth-token-env are mutually exclusive")
    if args.auth_token_env:
        value = os.environ.get(args.auth_token_env)
        if not value:
            raise ValueError(f"environment variable {args.auth_token_env!r} is not set")
        return value
    return args.auth_token


def _emit(summary: dict[str, Any]) -> None:
    sys.stdout.buffer.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    sys.stdout.buffer.write(b"\n")


def _close_client(client: object | None) -> None:
    http_client = getattr(client, "_http", None)
    close = getattr(http_client, "close", None)
    if callable(close):
        close()


def _set_timeout(client: object, timeout: float) -> None:
    http_client = getattr(client, "_http", None)
    if http_client is None:
        return
    try:
        http_client.timeout = httpx.Timeout(timeout)
    except Exception:
        return


def _base_summary(*, args: argparse.Namespace, sglang_url: str, actor: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "sglang_url": sglang_url,
        "model_short_name": args.model,
        "actor_checkpoint": actor,
        "d_model": None,
        "response_status": None,
        "raw_text_returned": False,
        "raw_text_prefix": None,
        "error_type": None,
        "error": None,
    }


def _run(args: argparse.Namespace) -> int:
    try:
        cfg = _load_yaml(Path(args.config))
        model_cfg = _resolve_model(cfg, args.model)
        decode_cfg = cfg.get("nla_decode", {})
        if not isinstance(decode_cfg, dict):
            raise ValueError("config nla_decode must be a mapping")
        sglang_url = args.sglang_url or str(decode_cfg.get("sglang_url", "http://localhost:30000"))
        auth_token = _resolve_auth_token(args)
        actor = str(model_cfg["nla_actor"])
        expected_d_model = int(model_cfg["d_model"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        summary = _base_summary(args=args, sglang_url=args.sglang_url or "unknown")
        summary.update({"error_type": exc.__class__.__name__, "error": str(exc)})
        _emit(summary)
        return 2

    summary = _base_summary(args=args, sglang_url=sglang_url, actor=actor)
    client = None
    try:
        checkpoint_path = resolve_checkpoint_path(actor)
        summary["actor_checkpoint"] = str(checkpoint_path)
        with redirect_stdout(sys.stderr):
            client = NLAClient(checkpoint_path, sglang_url=sglang_url, device="cpu")
        _set_timeout(client, float(args.timeout))
        apply_sglang_auth_token(client, auth_token)
    except Exception as exc:
        summary.update({"error_type": exc.__class__.__name__, "error": str(exc)})
        _emit(summary)
        return 3

    try:
        d_model = int(client.cfg.d_model)
        summary["d_model"] = d_model
        if d_model != expected_d_model:
            summary.update(
                {
                    "error_type": "DModelMismatch",
                    "error": f"client d_model={d_model} != config d_model={expected_d_model}",
                }
            )
            _emit(summary)
            return 5

        if args.skip_generate:
            summary["ok"] = True
            _emit(summary)
            return 0

        activation = np.full((d_model,), float(args.activation_value), dtype=np.float32)
        try:
            with redirect_stdout(sys.stderr):
                raw_text = client.generate(
                    activation,
                    extract_explanation=False,
                    temperature=0.0,
                    max_new_tokens=8,
                )
        except httpx.HTTPStatusError as exc:
            summary.update(
                {
                    "response_status": exc.response.status_code,
                    "error_type": "HTTPStatusError",
                    "error": str(exc),
                }
            )
            _emit(summary)
            return 4
        except httpx.HTTPError as exc:
            summary.update({"error_type": exc.__class__.__name__, "error": str(exc)})
            _emit(summary)
            return 4
        except (KeyError, TypeError, ValueError) as exc:
            summary.update({"error_type": exc.__class__.__name__, "error": str(exc)})
            _emit(summary)
            return 5

        if not isinstance(raw_text, str):
            summary.update(
                {
                    "error_type": "BadGenerateResponse",
                    "error": f"NLAClient.generate returned {type(raw_text).__name__}, expected str",
                }
            )
            _emit(summary)
            return 5
        summary["response_status"] = 200
        summary["raw_text_returned"] = bool(raw_text.strip())
        summary["raw_text_prefix"] = raw_text[:200]
        if not raw_text.strip():
            summary.update({"error_type": "EmptyGenerateOutput", "error": "generate returned empty text"})
            _emit(summary)
            return 5

        summary["ok"] = True
        _emit(summary)
        return 0
    finally:
        _close_client(client)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SGLang readiness for MedNLA AV decoding.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Model short_name from config.")
    parser.add_argument("--sglang-url", default=None, help="Override config nla_decode.sglang_url.")
    parser.add_argument("--auth-token", default=None, help="Bearer token for SGLang API auth. Prefer --auth-token-env.")
    parser.add_argument("--auth-token-env", default=None, help="Environment variable containing SGLang bearer token.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds for the readiness request.")
    parser.add_argument("--skip-generate", action="store_true", help="Only load the NLA client and validate d_model.")
    parser.add_argument("--activation-value", type=float, default=0.0, help="Value for the synthetic activation vector.")
    return _run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
