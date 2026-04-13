def process_order(order, user, is_admin=False):
    """Sample file for DefectSense prediction demo."""
    total = 0
    discount = 0

    if order is None:
        return {"status": "error", "reason": "missing order"}

    for item in order.get("items", []):
        if item.get("qty", 0) > 0:
            total += item.get("price", 0) * item.get("qty", 0)
        else:
            total += 0

    if user and user.get("tier") == "gold":
        discount += 0.10
    elif user and user.get("tier") == "silver":
        discount += 0.05
    elif is_admin:
        discount += 0.15

    if total > 1000:
        if user and user.get("country") == "IN":
            discount += 0.02
        else:
            discount += 0.01

    final_total = total - (total * discount)

    if final_total < 0:
        final_total = 0

    return {
        "status": "ok",
        "gross": round(total, 2),
        "discount": round(discount, 4),
        "net": round(final_total, 2),
    }
