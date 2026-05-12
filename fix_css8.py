import sys

filepath = 'src/app.py'
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_css_add = """
/* ── Dialog and Config Modals Light Mode Fix ── */
:root[data-theme="light"] dialog,
:root[data-theme="light"] .modal,
:root[data-theme="light"] .gr-dialog,
:root[data-theme="light"] [role="dialog"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
}
:root[data-theme="light"] dialog *,
:root[data-theme="light"] .modal *,
:root[data-theme="light"] .gr-dialog *,
:root[data-theme="light"] [role="dialog"] * {
    color: #0f172a !important;
}
:root[data-theme="light"] dialog button:hover,
:root[data-theme="light"] .modal button:hover,
:root[data-theme="light"] .gr-dialog button:hover,
:root[data-theme="light"] [role="dialog"] button:hover {
    background-color: #f1f5f9 !important;
}
"""
    
    css_start = content.find('CSS = """')
    if css_start != -1:
        css_end = content.find('"""', css_start + 10)
        if css_end != -1 and '/* ── Dialog and Config Modals Light Mode Fix ── */' not in content:
            content = content[:css_end] + new_css_add + content[css_end:]
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print('SUCCESS MODAL CSS')
        else:
            print('ALREADY APPLIED OR NOT FOUND CSS END')
    else:
        print('ERROR: CSS start not found')

except Exception as e:
    print('ERROR:', str(e))
