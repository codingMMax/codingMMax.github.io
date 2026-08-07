# latexmk configuration for this repository: force the XeLaTeX engine
# (the CV uses fontspec/Charter, which requires XeTeX or LuaTeX)
$pdf_mode = 1;
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder %O %S';