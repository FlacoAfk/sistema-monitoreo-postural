import sys

filepath = 'src/app.py'
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Reemplazar la creacion del Markdown
    old_markdown = 'model_info = gr.Markdown(t0["model_info_def"])'
    new_markdown = 'model_info = gr.Markdown(t0["model_info_def"], elem_classes=["center-text"])'
    if old_markdown in content:
        content = content.replace(old_markdown, new_markdown)

    # Agregar la clase CSS
    new_css = """
/* ── Text Alignment ── */
.center-text, .center-text p, .center-text .prose, .center-text .gr-markdown {
    text-align: center !important;
}
"""
    if '/* ── Text Alignment ── */' not in content:
        css_start = content.find('CSS = """')
        if css_start != -1:
            css_end = content.find('"""', css_start + 10)
            if css_end != -1:
                content = content[:css_end] + new_css + content[css_end:]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS MARKDOWN CENTER')

except Exception as e:
    print('ERROR:', str(e))
