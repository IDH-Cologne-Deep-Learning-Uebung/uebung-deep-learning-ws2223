with open("exercise-04\wiki.txt", "r") as f:
    with open("exercise-04\short.txt", "w") as f1:
        with open("exercise-04\ articles.txt", "w") as f2:
            with open("exercise-04\ april.txt", "w") as f3:
                for line in f:
                    if len(line) <= 50: 
                        f1.write(line)
                    if line.startswith("Die") or line.startswith("Der") or line.startswith("Das"):
                        f2.write(line)
                    #if "April" in f.read():
                     #   f3.write(line) funktioniert irgendwie nicht
                    else: 
                        print(None)