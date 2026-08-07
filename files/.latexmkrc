# latexmk configuration for the CV: force the XeLaTeX engine
# (fontspec/Charter requires XeTeX or LuaTeX)
$pdf_mode = 1;
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder %O %S';
