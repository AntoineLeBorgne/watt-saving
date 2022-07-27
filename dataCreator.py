# agent 0

fichier1 = open("data/train-watt-saving-agent0.csv", "a")
fichier2 = open("data/test-watt-saving-agent0.csv", "a")
fichier1.truncate(0)
fichier2.truncate(0)

fichier1.write("\"sequence\"\n")
fichier2.write("\"sequence\"\n")
for _ in range(100000):
    fichier1.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "1,1,0,0,1,0,0,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "1,1,0,0,1,0,0,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,1,0,0,0,0;0,1,0,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,1,0,0,0,0;0,1,0,0,0" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier2.write("\n")
fichier1.close()
fichier2.close()

# agent 1

fichier1 = open("data/train-watt-saving-agent1.csv", "a")
fichier2 = open("data/test-watt-saving-agent1.csv", "a")
fichier1.truncate(0)
fichier2.truncate(0)

fichier1.write("\"sequence\"\n")
fichier2.write("\"sequence\"\n")
for _ in range(100000):
    fichier1.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,1,0,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,1,0,0,0" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "1,1,0,0,0,1,0,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "1,1,0,0,0,1,0,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,1,0,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,1,0,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier2.write("\n")
fichier1.close()
fichier2.close()

# agent 2

fichier1 = open("data/train-watt-saving-agent2.csv", "a")
fichier2 = open("data/test-watt-saving-agent2.csv", "a")
fichier1.truncate(0)
fichier2.truncate(0)

fichier1.write("\"sequence\"\n")
fichier2.write("\"sequence\"\n")
for _ in range(100000):
    fichier1.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,1,0,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,1,0,0,0,0,0;0,1,0,0,0" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "1,1,0,0,0,0,1,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "1,1,0,0,0,0,1,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,1,0,0" + "\"")
    fichier2.write("\n")

    fichier1.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,0,1,1" + "\"")
    fichier1.write("\n")
    fichier2.write("\"" + "0,0,0,0,0,0,1,0,0,0;0,0,0,1,1" + "\"")
    fichier2.write("\n")
fichier1.close()
fichier2.close()
