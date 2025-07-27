#!/usr/bin/env python3
import sqlite3
import argparse
import os
import sys

def dump_sqlite_db(db_path, minimize):
    if not os.path.isfile(db_path):
        print(f"❌ Fichier introuvable : {db_path}", file=sys.stderr)
        sys.exit(1)

    minimize = bool(int(minimize))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Lister les tables
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [row[0] for row in cursor.fetchall()]
    if tables:
        print(f"🔍 Tables dans {db_path}:")
        for tbl in tables:
            print(f" - {tbl}")
        print()
    else:
        print("⚠️  Aucune table trouvée dans la base.")
        conn.close()
        return

    # 2. Affichage sommaire du contenu de chaque table
    for tbl in tables:
        print(f"=== Table `{tbl}` ===")
        cursor.execute(f"PRAGMA table_info({tbl});")
        cols = [col[1] for col in cursor.fetchall()]
        print("Colonnes :", ", ".join(cols))

        cursor.execute(f"SELECT * FROM {tbl};")
        rows = cursor.fetchall()
        if not rows:
            print("(vide)\n")
            continue

        # calcul des largeurs de colonnes
        widths = [len(c) for c in cols]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        # affiche tableau
        header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
        sep    = "-+-".join("-" * widths[i]      for i in range(len(cols)))
        print(header)
        print(sep)
        for row in rows:
            line = " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
            print(line)
        print()

    # 3. Meilleur trial global
    def get_best(where_clause="", params=(), minimize=False):

        if minimize : 
            query = (
                "SELECT trial_id, value "
                "FROM trial_values "
                f"{where_clause} "
                "ORDER BY value ASC "
                "LIMIT 1;"
            )
        else : 
            query = (
                "SELECT trial_id, value "
                "FROM trial_values "
                f"{where_clause} "
                "ORDER BY value DESC "
                "LIMIT 1;"
            )
        cursor.execute(query, params)
        return cursor.fetchone()

    # a) global
    best = get_best(minimize=minimize)
    if best:
        best_id, best_value = best
        print(f"🏆 Meilleur trial global : ID={best_id}  value={best_value}")
        cursor.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?;",
            (best_id,)
        )
        params = cursor.fetchall()
        if params:
            print("   Paramètres associés :")
            for name, val in params:
                print(f"    • {name} = {val}")
        else:
            print("   ⚠️ Aucun paramètre trouvé pour ce trial.")
    else:
        print("⚠️  Table trial_values vide ou inexistante (global).")

    print()

    # b) meilleur trial avec trial_id <= 100
    limited_best = get_best("WHERE trial_id <= ?", (100,),minimize=minimize)
    if limited_best:
        lid, lval = limited_best
        print(f"🎯 Meilleur trial avec trial_id ≤ 100 : ID={lid}  value={lval}")
        cursor.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?;",
            (lid,)
        )
        lparams = cursor.fetchall()
        if lparams:
            print("   Paramètres associés :")
            for name, val in lparams:
                print(f"    • {name} = {val}")
        else:
            print("   ⚠️ Aucun paramètre trouvé pour ce trial.")
    else:
        print("⚠️  Aucun trial avec trial_id ≤ 100 trouvé dans trial_values.")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Affiche le contenu d'un fichier SQLite et identifie les meilleurs trials."
    )
    parser.add_argument(
        "dbfile",
        help="Chemin vers le fichier .db ou .sqlite",
    )
    parser.add_argument(
        "min",
        help="Specify if best if the min or not",
    )
    
    args = parser.parse_args()
    
    dump_sqlite_db(args.dbfile,args.min)

if __name__ == "__main__":
    main()
