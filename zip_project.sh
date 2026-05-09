#!/bin/bash
OUTPUT_FILE="llm-proxy.zip"

# Eliminar zip anterior si existe
rm -f "$OUTPUT_FILE"

# Lista de archivos y directorios a incluir
ITEMS=(
    .claude
    agent
    app
    docs
    tests
    .env
    .env.example
    CLAUDE.md
    client.py
    main.py
    mcp.json
    notes.md
    README.md
    requirements.txt
    zip_project.sh
)

# Filtrar solo los que existen
EXISTING=()
for item in "${ITEMS[@]}"; do
    if [ -e "$item" ]; then
        EXISTING+=("$item")
    else
        echo "Advertencia: '$item' no encontrado, se omite."
    fi
done

zip -r "$OUTPUT_FILE" "${EXISTING[@]}" \
    -x "*/__pycache__/*" \
    -x "*/*.pyc"

echo "Proyecto comprimido en $OUTPUT_FILE"
