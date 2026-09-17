python set_clients.py --path /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/dataset/NEU_721/data.yaml --clients 10 --seed 2508 > log.txt 2>&1
python cam.py > log.txt 2>&1
python cam_inpainting.py > log.txt 2>&1
for i in $(seq 1 10); do
    echo "CLIENT = $i"
    echo "IMAGE = /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}/train/images"
    echo "MASK  = /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/attention_mask/client_${i}/train/images"
    echo "OUTPUT = /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}_inpainting_reconstructed/train/images"
    echo
    python -m scripts.demo \
        --model-name migan-512 \
        --model-path ./models/migan_512_places2.pt \
        --images-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}/train/images \
        --masks-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/attention_mask/client_${i}/train/images \
        --output-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}_inpainting_reconstructed/train/images \
        --device cuda \
        --invert-mask 
    python -m scripts.demo \
        --model-name migan-512 \
        --model-path ./models/migan_512_places2.pt \
        --images-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}/val/images \
        --masks-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/attention_mask/client_${i}/val/images \
        --output-dir /home/gssi-lab/Documents/trieu/4_10_clients/Ours_wi_SO_wi_inpainting_NEU/client_${i}_inpainting_reconstructed/val/images \
        --device cuda \
        --invert-mask 
done > log.txt 2>&1
python clients_studies.py > log.txt 2>&1
python train_4_10_clients_two_stage_all_rounds_v3.py > log.txt 2>&1
python test.py > log_test.txt 2>&1
