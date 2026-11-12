# Shemfile highlighting for VS Code

Highlights `Shemfile`s with VS Code's Dockerfile grammar, plus Shem's own
`WARM` and `KERNEL` instructions. Shemfiles get their own `shemfile` language,
so Dockerfile linters don't flag those instructions as unknown. A scroll icon
(the shem, a slip of words) marks them in file icon themes, like the default
Seti, that have no icon of their own for the language.

VS Code offers it as a workspace extension: in the Extensions view, search
`@recommended` and install *Shemfile* under Workspace Recommendations. It
installs for this workspace only, once the workspace is trusted. (VS Code 1.88
had this behind `extensions.experimental.supportWorkspaceExtensions`; current
versions have no such setting.)

Or install it from a package:

```sh
cd .vscode/extensions/shemfile
npx @vscode/vsce package
code --install-extension shemfile-0.0.1.vsix
```
