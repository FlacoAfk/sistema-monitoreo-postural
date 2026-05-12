import sys

filepath = 'src/app.py'
try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Dropdown: quitar container=False, ocultar label, arreglar colores menu
    if 'container=False,' in content:
        content = content.replace('container=False,', 'container=True, show_label=False,')

    # 2. Agregar CSS para el menu desplegable de opciones y parpadeo de video
    new_css_add = """
/* ── Dropdown List Light Mode Fix ── */
:root[data-theme="light"] .gr-dropdown-list,
:root[data-theme="light"] ul.options,
:root[data-theme="light"] .options {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}
:root[data-theme="light"] .gr-dropdown-list li,
:root[data-theme="light"] ul.options li,
:root[data-theme="light"] .options li {
    color: #0f172a !important;
}
:root[data-theme="light"] .gr-dropdown-list li:hover,
:root[data-theme="light"] ul.options li:hover,
:root[data-theme="light"] .options li:hover,
:root[data-theme="light"] .gr-dropdown-list li.selected,
:root[data-theme="light"] ul.options li.selected {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}

/* ── Prevent Video Flicker ── */
.pm-leftcol .image-frame,
.pm-leftcol .image-container,
.pm-leftcol img,
.pm-leftcol video,
.pm-leftcol [data-testid="image"] {
    background-color: transparent !important;
    transition: none !important;
    animation: none !important;
}
"""
    
    css_start = content.find('CSS = """')
    if css_start != -1:
        css_end = content.find('"""', css_start + 10)
        if css_end != -1 and '/* ── Dropdown List Light Mode Fix ── */' not in content:
            content = content[:css_end] + new_css_add + content[css_end:]
            
    # 3. Arreglar la alineacion en Top Tools
    # Reemplazamos la alineacion estricta si existe
    align_fix = """/* ── Top Tools Fix Alineacion Estricta ── */
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
}"""

    new_align_fix = """/* ── Top Tools Alineacion Nativa ── */
.top-tools {
    align-items: center !important;
    justify-content: flex-end !important;
}
#pm-theme-toggle {
    margin: 0 !important;
    position: static !important;
    height: 42px !important;
}"""

    if align_fix in content:
        content = content.replace(align_fix, new_align_fix)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')

except Exception as e:
    print('ERROR:', str(e))
