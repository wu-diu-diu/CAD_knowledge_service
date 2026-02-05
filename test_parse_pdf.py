import argparse
import asyncio
from pathlib import Path
import os
from main import BASE_DIR, parse_pdf

os.environ.setdefault("VLLM_USE_V1", "0")

async def run(pdf_path: Path) -> None:
    await parse_pdf(pdf_path)
    output_dir = BASE_DIR / "transfer_output"
    if not output_dir.exists():
        print("transfer_output directory not found.")
        return
    candidates = list(output_dir.glob("*.md"))
    if not candidates:
        print("No markdown output found.")
        return
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Saved markdown: {latest}")
    preview = "\n".join(latest.read_text(encoding="utf-8", errors="ignore").splitlines()[:60])
    print("\n--- Preview ---\n")
    print(preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test parse_pdf with page markers.")
    parser.add_argument("pdf_path", type=Path, help="Path to the input PDF file.")
    args = parser.parse_args()
    asyncio.run(run(args.pdf_path))


if __name__ == "__main__":
    main()
