import sys

filepath = 'src/app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_css = """
/* ── Top Tools Fix Alineacion Estricta ── */
.top-tools {
    align-items: flex-end !important;
}
.top-tools > .col {
    padding-bottom: 0px !important;
    margin-bottom: 0px !important;
    align-self: flex-end !important;
}
.top-tools .gr-dropdown {
    margin-bottom: 0px !important;
    padding-bottom: 0px !important;
}
#pm-theme-toggle {
    margin-bottom: 0px !important;
    margin-top: auto !important; /* Forza ir al fondo */
    position: relative !important;
    top: 4px !important; /* Ajuste manual hacia abajo para igualar visualmente el input que no tiene borde */
}
"""

if '/* ── Top Tools Fix Alineacion Estricta ── */' not in content:
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
    print('ALREADY APPLIED')
