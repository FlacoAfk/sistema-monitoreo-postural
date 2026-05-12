import sys

filepath = 'src/app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Modify the Row to add elem_classes
old_row = '        # ── Selector de idioma + toggle tema (top-right) ──\n        with gr.Row():'
new_row = '        # ── Selector de idioma + toggle tema (top-right) ──\n        with gr.Row(elem_classes=["top-tools"]):'
if old_row in content:
    content = content.replace(old_row, new_row)
else:
    print('Row not found or already modified')

# 2. Add CSS
new_css = """
/* ── Top Tools (Theme & Lang) ── */
.top-tools {
    align-items: flex-end !important;
}
#pm-theme-toggle {
    background: var(--pm-surface-2) !important;
    border: 1px solid var(--pm-border) !important;
    border-radius: var(--pm-radius-sm) !important;
    color: var(--pm-text-1) !important;
    height: 42px !important;
    width: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 18px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
    margin-bottom: 2px !important;
}
#pm-theme-toggle:hover {
    background: var(--pm-surface) !important;
    border-color: var(--pm-border-strong) !important;
    transform: translateY(-1px) !important;
}
:root[data-theme="light"] #pm-theme-toggle {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}
:root[data-theme="light"] #pm-theme-toggle:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
}
"""

if '/* ── Top Tools (Theme & Lang) ── */' not in content:
    css_start = content.find('CSS = """')
    if css_start != -1:
        css_end = content.find('"""', css_start + 10)
        if css_end != -1:
            content = content[:css_end] + new_css + content[css_end:]
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print('SUCCESS CSS')
        else:
            print('ERROR: CSS end not found')
    else:
        print('ERROR: CSS start not found')
else:
    print('CSS ALREADY APPLIED')
