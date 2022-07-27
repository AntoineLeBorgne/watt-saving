nb_agent = 3
nb_iteration = 100000
for toto in range(nb_agent):
    fichier = open("train-watt-saving-agent"+str(toto)+".csv", "a")
    fichier.truncate(0)
    fichier.write("\"sequence\"\n")
    for _ in range(nb_iteration):
        for agent in range(nb_agent):
            fichier.write("\"")
            # demande et offre proposé a l'agent
            for _ in range(nb_agent-1):
                fichier.write("0,0,")
            # recommandation de l'environnement pour les tour de garde
            truc = agent
            cpt = 1
            while truc != 0:
                fichier.write("0,")
                cpt += 1
                truc -= 1
            fichier.write("1,")
            for _ in range(nb_agent-cpt):
                fichier.write("0,")
            # les agents blacklist
            for tata in range(nb_agent):
                if tata == nb_agent-1:
                    fichier.write("0;")
                else:
                    fichier.write("0,")
            # l'action correspondant a l'état
            # le param si l'agent ne fait rien
            fichier.write("0,")
            # les demande faite a l'agent
            if toto == agent:
                for _ in range(nb_agent - 1):
                    fichier.write("0,")
                for tyty in range(nb_agent - 1):
                    if tyty == nb_agent-2:
                        fichier.write("1")
                    else:
                        fichier.write("1,")
            else:
                if toto > agent:
                    cpt = 0
                    for _ in range(nb_agent - 1):
                        if cpt == agent:
                            fichier.write("1,")
                        else:
                            fichier.write("0,")
                        cpt += 1
                if toto < agent:
                    cpt = 0
                    for _ in range(nb_agent):
                        if cpt == toto:
                            cpt += 1
                        else:
                            if cpt == agent:
                                fichier.write("1,")
                            else:
                                fichier.write("0,")
                            cpt += 1
                # les offres faites a l'agent
                for tete in range(nb_agent - 1):
                    if tete == nb_agent-2:
                        fichier.write("0")
                    else:
                        fichier.write("0,")
            fichier.write("\"\n")

        # deuxieme etat concernant celui qui recois
        # demande et offre proposé a l'agent
        fichier.write("\"")
        for _ in range(nb_agent - 1):
            fichier.write("1,")
        for _ in range(nb_agent - 1):
            fichier.write("0,")
        # recommandation de l'environnement pour les tour de garde
        truc = toto
        cpt = 1
        while truc != 0:
            fichier.write("0,")
            cpt += 1
            truc -= 1
        fichier.write("1,")
        for _ in range(nb_agent - cpt):
            fichier.write("0,")
        # les agents blacklist
        for tata in range(nb_agent):
            if tata == nb_agent - 1:
                fichier.write("0;")
            else:
                fichier.write("0,")
        # l'action correspondant a l'état
        # le param si l'agent ne fait rien
        fichier.write("0,")
        # les demande faite a l'agent
        for _ in range(nb_agent - 1):
            fichier.write("0,")
        # les offres faites a l'agent
        for titi in range(nb_agent - 1):
            if titi == nb_agent - 2:
                fichier.write("1")
            else:
                fichier.write("1,")
        fichier.write("\"\n")
    fichier.close()
