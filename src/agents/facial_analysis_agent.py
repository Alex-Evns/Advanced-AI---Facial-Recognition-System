def analyse_face(predictions):

    glasses = predictions["glasses"]
    hat = predictions["hat"]
    young = predictions["young"]

    summary = []

    # Glasses

    if glasses >= 0.5:

        summary.append(
            f"The subject is likely wearing glasses ({glasses:.1%} confidence)."
        )

    else:

        summary.append(
            f"The subject is unlikely to be wearing glasses ({1-glasses:.1%} confidence)."
        )

    # Hat

    if hat >= 0.5:

        summary.append(
            f"The subject is likely wearing a hat ({hat:.1%} confidence)."
        )

    else:

        summary.append(
            f"No hat was detected ({1-hat:.1%} confidence)."
        )

    # Young

    if young >= 0.5:

        summary.append(
            f"The subject appears young ({young:.1%} confidence)."
        )

    else:

        summary.append(
            f"The subject does not appear young ({1-young:.1%} confidence)."
        )

    return "\n".join(summary)