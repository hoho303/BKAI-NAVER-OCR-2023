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

# master
curl -L -o master_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1TCEAal45DuwKPTMphs32N-Yo8ZwlVyUN&export=download&confirm=t'

# satrn
curl -L -o satrn_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1CtKhw3MLacf08-uK5fUNEalysJDOW7Yw&export=download&confirm=t'

# parseq
curl -L -o parseq_100_0.ckpt 'https://drive.google.com/uc?export=download&id=1HyJ-nt14TGaDviCuPLtdAfrLYUwVJU1O&export=download&confirm=t'

# vietocr
curl -L -o 'transformerocr.pth' 'https://drive.google.com/uc?export=download&id=1JIEBvWOYRjRhz-1cCRUo1ozV8b_6mrVT&export=download&confirm=t'