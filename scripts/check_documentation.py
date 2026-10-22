"""Audit the Markdown actually published by Studio, without starting any job."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from urllib.parse import unquote,urlsplit


def published_paths(root):
    from studio.documents import aliases
    mapping=aliases(root)
    paths=list(root.glob('*.md'))+list((root/'docs').rglob('*.md'))+list((root/'skills').rglob('*.md'))+list((root/'go2_nav/paw_contact').glob('*.md'))
    if (root/'studio/README.md').exists():paths.append(root/'studio/README.md')
    return sorted(p for p in set(paths) if p.relative_to(root).as_posix() not in mapping or not (root/mapping[p.relative_to(root).as_posix()]).exists())


def source_options(root,module):
    path=root.joinpath(*module.split('.')).with_suffix('.py')
    if not path.is_file():path=root.joinpath(*module.split('.'))/'__main__.py'
    if not path.is_file():return None
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    return {a.value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='add_argument' for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')}


def audit(root,bash=None):
    root=Path(root).resolve();issues=[];checked_commands=0;blocks=0;files=published_paths(root)
    app_flags={'--headless','--device','--enable_cameras','--kit_args'}
    upstream_backflip_flags={'--trainer','--num_envs','--max_iterations','--seed','--reset_optimizer','--min_action_std','--max_action_std','--resume','--checkpoint','--safety_warmup_iterations','--safety_ramp_iterations'}
    for path in files:
        rel=path.relative_to(root).as_posix();text=path.read_text(encoding='utf-8-sig')
        def issue(kind,message):issues.append(dict(path=rel,kind=kind,message=message))
        if re.search(r'/root(?:/|\b)|/home/[^\s/]+/|/workspace/|[XC]:\\(?:Items|Users)\\',text):issue('machine_path','Contains a machine-specific home/workspace path')
        for match in re.finditer(r'\]\(([^\s)]+)\)',text):
            href=unquote(match[1]);u=urlsplit(href)
            if u.scheme or href.startswith(('#','/api/')):continue
            target=(path.parent/u.path).resolve()
            if not u.path or '${' in u.path or '$' in u.path or '<' in u.path:continue
            if not target.is_relative_to(root):issue('link_escape',href)
            elif not target.exists():issue('missing_link',href)
        for match in re.finditer(r'^```(bash|sh|shell)\s*\n(.*?)^```\s*$',text,re.M|re.S):
            script=match[2];blocks+=1
            if re.search(r'<[\w\u4e00-\u9fff][^<>\n]*>',script):issue('shell_placeholder','Replace angle-bracket placeholders with defined, quoted environment variables')
            if bash:
                r=subprocess.run([bash,'-n'],input=script,text=True,capture_output=True)
                if r.returncode:issue('bash_syntax',r.stderr.strip()[:300])
            primary=path.parent.parent.name=='skills' and path.name in ('训练手册.md','仿真手册.md','部署手册.md')
            if not primary:continue
            for line in script.replace('\\\n',' ').splitlines():
                if re.match(r'^\s*\+\s+--',line):issue('stray_patch_marker',line)
                try:tokens=shlex.split(line)
                except ValueError as e:issue('shell_tokens',str(e));continue
                if '-m' not in tokens:continue
                i=tokens.index('-m')
                if i+1>=len(tokens):continue
                module=tokens[i+1]
                if not module.startswith(('skills.','rob2nav.','moe_perceptive.','go2_nav.','studio.')):continue
                options=source_options(root,module)
                if options is None:issue('missing_module',module);continue
                allowed=options|app_flags
                if module=='skills.backflip.train_isaac51':allowed|=upstream_backflip_flags
                for flag in tokens[i+2:]:
                    if flag.startswith('--') and flag.split('=')[0] not in allowed:issue('unknown_option',module+' '+flag)
                checked_commands+=1
    return dict(files=len(files),bash_blocks=blocks,primary_commands_checked=checked_commands,
                bash_parser=bool(bash),issues=issues,passed=not issues,
                scope='Published source Markdown: local links, portable paths, Bash syntax, primary manual CLI option names. No GPU training, ROS navigation, or hardware replay is implied.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);p.add_argument('--bash');a=p.parse_args()
    root=a.root.resolve()
    import sys
    sys.path.insert(0,str(root))
    git_bash=Path(os.environ.get('ProgramFiles','C:/Program Files'))/'Git/bin/bash.exe'
    bash=a.bash or (str(git_bash) if git_bash.is_file() else shutil.which('bash'))
    result=audit(root,bash)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
