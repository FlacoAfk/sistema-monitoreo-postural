import sys

filepath = 'src/app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_css = """
/* ── Light mode: Forzar reset en contenedores nativos ── */
:root[data-theme="light"] .pm-leftcol .block,
:root[data-theme="light"] .pm-leftcol .gr-block,
:root[data-theme="light"] .pm-leftcol .gr-box,
:root[data-theme="light"] .pm-leftcol .gr-panel,
:root[data-theme="light"] .gr-accordion,
:root[data-theme="light"] .gr-dropdown {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}

:root[data-theme="light"] .gr-dropdown select,
:root[data-theme="light"] .gr-accordion > button {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

:root[data-theme="light"] .gr-block > label, 
:root[data-theme="light"] .gr-label label, 
:root[data-theme="light"] span.label, 
:root[data-theme="light"] .block-title { 
    color: #4f46e5 !important;
    background: #f1f5f9 !important;
}
"""

if '/* ── Light mode: Forzar reset en contenedores nativos ── */' not in content:
    # Find the CSS block
    css_start = content.find('CSS = """')
    if css_start != -1:
        css_end = content.find('"""', css_start + 10)
        if css_end != -1:
            content = content[:css_end] + new_css + content[css_end:]
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print('SUCCESS')
        else:
            print('ERROR: CSS end not found')
    else:
        print('ERROR: CSS start not found')
else:
    print('ALREADY_APPLIED')
