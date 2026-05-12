import sys

filepath = 'src/app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_css = """
/* ── Light mode: Forzar textos legibles ── */
:root[data-theme="light"] .pm-leftcol,
:root[data-theme="light"] .pm-leftcol .markdown, 
:root[data-theme="light"] .pm-leftcol .gr-markdown, 
:root[data-theme="light"] .pm-leftcol .prose, 
:root[data-theme="light"] .pm-leftcol p,
:root[data-theme="light"] .pm-leftcol td,
:root[data-theme="light"] .pm-leftcol th,
:root[data-theme="light"] .pm-leftcol span,
:root[data-theme="light"] .gr-accordion,
:root[data-theme="light"] .gr-accordion .markdown,
:root[data-theme="light"] .gr-accordion p,
:root[data-theme="light"] .gr-accordion td,
:root[data-theme="light"] .gr-accordion th,
:root[data-theme="light"] .gr-form-info,
:root[data-theme="light"] .gr-text-sm,
:root[data-theme="light"] span[data-testid="block-info"] {
    color: #0f172a !important;
}

:root[data-theme="light"] .pm-leftcol .gr-form-info,
:root[data-theme="light"] .pm-leftcol span.text-sm,
:root[data-theme="light"] .gr-input-label span {
    color: #334155 !important;
}
"""

if '/* ── Light mode: Forzar textos legibles ── */' not in content:
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
