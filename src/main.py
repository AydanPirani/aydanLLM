from dataload.dataloader import create_dataloader

def main():
    with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)



if __name__ == "__main__":
    main()