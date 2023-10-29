mkdir -p /workspace/checkpoints

# abinet1
cd /workspace/checkpoints

curl -L -o abinet1_80_20.ckpt 'https://drive.google.com/uc?export=download&id=10_Wq7yP6MCwvd6033akrxqNYgs6_fb4A&export=download&confirm=t'

# abinet2
curl -L -o abinet2_80_20.ckpt 'https://drive.google.com/uc?export=download&id=1-p8vGmRsXsz1t-GNHAHdTIqxNVr5MXiX&export=download&confirm=t'

# abinet3
curl -L -o abinet3_80_20.ckpt 'https://drive.google.com/uc?export=download&id=1aKzJUxbiZ7zeovyv6nGKr5xtG5txzp7a&export=download&confirm=t'

# abinet4
curl -L -o abinet4_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1VkeycE1zxMdSsOaaxX-YR1wwbycXhZgy&export=download&confirm=t'

# abinet 5
curl -L -o abinet5_80_20.ckpt 'https://drive.google.com/uc?export=download&id=1fLAtgrG225BdRU4yem5I0wc7U8KonVYi&export=download&confirm=t'

# abinet 6
curl -L -o abinet6_80_20.ckpt 'https://drive.google.com/uc?export=download&id=1dbGeJxHwbzNt8T0hH2glrhiF5kzBqWXs&export=download&confirm=t'

# master
curl -L -o master_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1TCEAal45DuwKPTMphs32N-Yo8ZwlVyUN&export=download&confirm=t'

# satrn
curl -L -o satrn_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1CtKhw3MLacf08-uK5fUNEalysJDOW7Yw&export=download&confirm=t'

# parseq
curl -L -o parseq_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1HyJ-nt14TGaDviCuPLtdAfrLYUwVJU1O&export=download&confirm=t'

# vietocr
curl -L -o 'vietocr_100_0.pth' 'https://drive.google.com/uc?export=download&id=1Cn-cjy45YMGzoYTUyJ6NUJp5Wpi2mQiF&export=download&confirm=t'

# corner transformer
curl -L -o 'corner_100_0.pth' 'https://drive.google.com/uc?export=download&id=1mBN4PuIqDkU6qsHnNQlhNxWp8Y4R6Ol3&export=download&confirm=t'