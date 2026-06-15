def analyse_face(predictions):

    glasses = predictions["glasses"]
    hat = predictions["hat"]
    young = predictions["young"]

    summary = []


    # Glasses

    if glasses >= 0.5:

        summary.append(
            f"Glasses detected ({glasses:.1%} confidence)."
        )

    else:

        summary.append(
            f" No Glasses detected ({1-glasses:.1%} confidence)."
        )

    # Hat

    if hat >= 0.5:

        summary.append(
            f"Hat detected ({hat:.1%} confidence)."
        )

    else:

        summary.append(
            f" No Hat detected ({1-hat:.1%} confidence)."
        )


    # Young

    if young >= 0.5:

        summary.append(
            f"Young detected ({young:.1%} confidence)."
        )

    else:

        summary.append(
            f"Not young detected ({1-young:.1%} confidence)."
        )

    return "\n".join(summary)