# bid-engine

AI-powered tool that turns marked-up construction plan PDFs into priced bid spreadsheets.

## Install

```sh
make install
```

Requires Python 3.10+ and `poppler` (for `pdf2image`). On macOS:

```sh
brew install poppler
```

Set your Anthropic API key:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```sh
python scripts/bid.py plan.pdf --pages 2,3,5,6 --state NY --output bid.xlsx
```

Optional flags: `--stories <N>`, `--name <project name>`, `--address <addr>`, `--date YYYY-MM-DD`, `--quiet`.

## Test

```sh
make test
```

85 tests passing.
