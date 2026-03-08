import lseg.data as ld

ld.open_session()

for ric in ["SPX1MO=R", "SPX3MO=R", "SPX6MO=R", "SPX12MO=R"]:
    try:
        df = ld.get_data(
            universe=[ric],
            fields=["CF_LAST", "CF_CLOSE", "DSPLY_NAME"]
        )
        print(f"\n=== {ric} ===")
        print(df)
    except Exception as e:
        print(f"\n=== {ric} ===")
        print("ERROR:", e)

ld.close_session()