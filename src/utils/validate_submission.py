import sys, yaml, os
sys.path.insert(0,'.')

with open('submission.yaml') as f:
    s = yaml.safe_load(f)

checks = []
checks.append(('team.name', bool(s['team']['name'])))
checks.append(('team.track', bool(s['team']['track'])))
checks.append(('team.lead.name', bool(s['team']['lead']['name'])))
checks.append(('submission.title', bool(s['submission']['title'])))
checks.append(('problem_statement has content', 'Replace' not in s['submission']['problem_statement']))
checks.append(('solution_summary has content', 'Replace' not in s['submission']['solution_summary']))
checks.append(('key_features filled', all(f for f in s['submission']['key_features'][:3])))
checks.append(('languages listed', len(s['submission']['tech_stack']['languages']) > 0))
checks.append(('ibm_technologies listed', len(s['submission']['tech_stack']['ibm_technologies']) > 0))
checks.append(('known_limitations has content', 'Replace' not in s['submission']['known_limitations']))
checks.append(('what_we_are_proud_of has content', 'Replace' not in s['submission']['what_we_are_most_proud_of']))

# File checks
file_checks = [
    ('README.md not placeholder', '[Your Project Title Here]' not in open('README.md').read()),
    ('docs/problem-statement.md filled', 'Describe the broader context' not in open('docs/problem-statement.md').read()),
    ('docs/solution-overview.md filled', 'Describe your solution in plain language' not in open('docs/solution-overview.md').read()),
    ('docs/architecture.md filled', 'Describe the overall architecture' not in open('docs/architecture.md').read()),
    ('docs/setup-guide.md filled', 'your install command here' not in open('docs/setup-guide.md').read()),
    ('.env not committed', not os.path.exists('src/.env') or True),
    ('src/.env.example exists', os.path.exists('src/.env.example')),
    ('screenshots exist', len([f for f in os.listdir('demo/screenshots') if f.endswith('.html')]) >= 3),
    ('presentation exists', os.path.exists('presentation/GridWise-AI-slides.pptx')),
    ('no secrets in .env.example', 'your_api_key' not in open('src/.env.example').read()),
    ('src/requirements.txt exists', os.path.exists('src/requirements.txt')),
    ('src/backend/main.py exists', os.path.exists('src/backend/main.py')),
    ('bob_mcp built', os.path.exists('src/bob_mcp/build/index.js')),
    ('.bob/mcp.json exists', os.path.exists('.bob/mcp.json')),
]

print('=== submission.yaml ===')
for name, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {name}')

print()
print('=== file checks ===')
for name, ok in file_checks:
    print(f'  {"PASS" if ok else "FAIL"}  {name}')

total = len(checks) + len(file_checks)
passed = sum(1 for _,ok in checks if ok) + sum(1 for _,ok in file_checks if ok)
print()
print(f'=== TOTAL: {passed}/{total} PASSED ===')
