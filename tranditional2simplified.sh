for file in 'ls *.json'
do
cconv -f utf8-tw -t UTF8-CN -o data/$file
done
