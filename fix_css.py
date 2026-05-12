import sys
import os

filepath = 'src/app.py'

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    replacements = [
        (
            'body, .gradio-container {\n    background: var(--pm-bg) !important;',
            'body, html, .gradio-container {\n    background: var(--pm-bg) !important;\n    background-color: var(--pm-bg) !important;'
        ),
        (
            ':root[data-theme="light"] .gradio-container,\n:root[data-theme="light"] body { background: var(--pm-bg) !important; color: var(--pm-text-1) !important; }',
            ':root[data-theme="light"] .gradio-container,\n:root[data-theme="light"] html,\n:root[data-theme="light"] body { background: var(--pm-bg) !important; background-color: var(--pm-bg) !important; color: var(--pm-text-1) !important; }'
        ),
        (
            '/* Accordion */\n.gr-accordion {\n    background: var(--pm-surface) !important;\n    border: 1px solid var(--pm-border) !important;\n    border-radius: var(--pm-radius-sm) !important;\n    margin-bottom: var(--pm-space-2) !important;\n}\n.gr-accordion > button {\n    color: var(--pm-text-2) !important;\n    font-weight: 600 !important;\n    font-size: 12px !important;\n    padding: 10px 14px !important;\n}',
            '/* Accordion */\n.gr-accordion {\n    background: var(--pm-surface-2) !important;\n    border: 1px solid var(--pm-border-strong) !important;\n    border-radius: var(--pm-radius-sm) !important;\n    margin-bottom: var(--pm-space-2) !important;\n    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;\n}\n.gr-accordion > button {\n    color: var(--pm-text-1) !important;\n    font-weight: 600 !important;\n    font-size: 12px !important;\n    padding: 10px 14px !important;\n    background: var(--pm-surface) !important;\n    border-radius: var(--pm-radius-sm) !important;\n}'
        )
    ]

    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
        else:
            print(f"No match for: {old[:30]}...")

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print('SUCCESS')
    else:
        print('NO_CHANGE')
except Exception as e:
    print(e)
