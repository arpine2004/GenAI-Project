#  Start the RAG Q&A Assistant backend server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".env" ]; then
  echo "No .env file found. Copying from .env.example..."
  cp .env.example .env
  echo "Please edit .env and set your ANTHROPIC_API_KEY, then re-run."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  echo "Installing dependencies (this may take a few minutes first time)..."
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt -q
  echo "Dependencies installed."
fi

echo ""
echo "Starting RAG Q&A Assistant..."
echo "API: http://localhost:8000"
echo "UI: http://localhost:8000/app"
echo "API Docs: http://localhost:8000/docs"
echo ""

.venv/bin/uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir backend
