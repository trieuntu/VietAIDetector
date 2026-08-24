#!/bin/bash
# Compile Graphviz architecture diagram to PDF, SVG, EPS, and PNG.

echo "Generating Architecture Diagrams..."

if ! command -v dot &> /dev/null; then
    echo "Error: 'dot' command could not be found. Please install graphviz."
    echo "Ubuntu/Debian: sudo apt install graphviz"
    echo "Mac/Homebrew: brew install graphviz"
    exit 1
fi

dot -Tpdf architecture.dot -o architecture.pdf
dot -Teps architecture.dot -o architecture.eps
dot -Tsvg architecture.dot -o architecture.svg
dot -Tpng architecture.dot -o architecture.png -Gdpi=300

echo "Done! Generated architecture.pdf, architecture.eps, architecture.svg, architecture.png."
