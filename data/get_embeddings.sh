npm install --global yarn
for item in en \fr \ar \bg \ca \hr \cs \da \nl \et \fi \de \el \he \hu \id \it \mk \no \pl \pt \ro \ru \sk \sl \es \sv \tr \uk \vi; do \
echo "Loading model for language: $item"; \
cd raw && curl -L -O https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.$item.vec --output wiki.multi.$item.vec; \
yarn run compress -- $item; \
rm -rf raw/wiki.multi.$item.vec; \
done;