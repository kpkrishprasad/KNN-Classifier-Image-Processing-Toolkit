def read_write(file_name):
    with open(file_name, 'r') as my_file:
        my_file = my_file.readlines()
    print(my_file)

    with open("scary_words.txt", 'w') as ur_file:
        for i in my_file:
            if i.strip("\n")[-1] == "!":
                ur_file.write(i)

read_write("test.txt")
