"""
Packaging Rules Lookup Matrix (sprs/rules.py)

Maps identified object categories to deterministic packaging specifications,
including cardboard grades, padding buffers, protective fill materials,
handling instructions, and orientation recommendations.
"""

from typing import Dict, Any

CATEGORIES = [
    "Electronics",
    "Mobile Devices/Earbuds",
    "Fragile (Glass/Ceramics)",
    "Glassware/Bottles",
    "Toys/Plastics",
    "Books/Stationery",
    "Clothing/Textiles",
    "Heavy Tools/Hardware",
    "Other/Miscellaneous"
]

RULES_TABLE: Dict[str, Dict[str, Any]] = {
    "Electronics": {
        "cardboard_type": "Double-Wall Heavy Duty Corrugated (32 ECT)",
        "cardboard_gsm": 320,
        "padding_buffer_cm": 2.5,
        "filling_material": "Anti-Static Bubble Wrap + High-Density Foam Inserts",
        "filling_density_g_cm3": 0.025,
        "handling_instructions": "FRAGILE — ESD SENSITIVE DEVICE — KEEP DRY & DO NOT STACK HEAVY LOADS",
        "orientation_recommendation": "Flat / Screen Face Up",
        "weight_factor_multiplier": 1.2,
        "item_density_g_cm3": 0.38,
        "eco_rating": "A",
        "co2_factor_g_per_cm3": 0.012,
        "badge_color": "#00f2fe",
        "warning_flags": ["ESD Sensitive", "Fragile Glass Screen", "Keep Dry"]
    },
    "Mobile Devices/Earbuds": {
        "cardboard_type": "Double-Wall Rigid Compact Kraft Box (32 ECT)",
        "cardboard_gsm": 300,
        "padding_buffer_cm": 2.0,
        "filling_material": "Anti-Static Bubble Pouch + Molded Paper Cushion",
        "filling_density_g_cm3": 0.022,
        "handling_instructions": "HIGH VALUE ELECTRONICS — LITHIUM BATTERY INSIDE — HANDLE WITH CARE",
        "orientation_recommendation": "Flat Face Up",
        "weight_factor_multiplier": 1.1,
        "item_density_g_cm3": 0.45,  # High precision density for lithium & battery cases
        "eco_rating": "A+",
        "co2_factor_g_per_cm3": 0.009,
        "badge_color": "#38bdf8",
        "warning_flags": ["Lithium Battery", "High Value", "ESD Sensitive"]
    },
    "Fragile (Glass/Ceramics)": {
        "cardboard_type": "Heavy Duty Double-Wall (44 ECT / Reinforced Kraft)",
        "cardboard_gsm": 420,
        "padding_buffer_cm": 4.0,
        "filling_material": "Pre-formed Thermocol (EPS) Blocks + Heavy Duty Bubble Wrap",
        "filling_density_g_cm3": 0.035,
        "handling_instructions": "FRAGILE — GLASSWARE/CERAMIC — THIS SIDE UP — HANDLE WITH EXTREME CARE",
        "orientation_recommendation": "Upright / Vertical Only",
        "weight_factor_multiplier": 1.5,
        "item_density_g_cm3": 0.65,
        "eco_rating": "B",
        "co2_factor_g_per_cm3": 0.018,
        "badge_color": "#ff4b2b",
        "warning_flags": ["Fragile Glass", "Handle With Care", "This Side Up"]
    },
    "Glassware/Bottles": {
        "cardboard_type": "Reinforced 44 ECT Heavy Duty Double-Wall",
        "cardboard_gsm": 450,
        "padding_buffer_cm": 4.5,
        "filling_material": "Inflatable Air Column Cushion + EPS Bottle Sleeve",
        "filling_density_g_cm3": 0.038,
        "handling_instructions": "FRAGILE LIQUID CONTAINER — THIS SIDE UP — DO NOT DROP",
        "orientation_recommendation": "Upright Vertical",
        "weight_factor_multiplier": 1.6,
        "item_density_g_cm3": 0.72,
        "eco_rating": "B",
        "co2_factor_g_per_cm3": 0.020,
        "badge_color": "#f43f5e",
        "warning_flags": ["Fragile Glass", "Liquid Vessel", "This Side Up"]
    },
    "Toys/Plastics": {
        "cardboard_type": "Single-Wall Corrugated (200 lb test / 32 ECT)",
        "cardboard_gsm": 210,
        "padding_buffer_cm": 1.5,
        "filling_material": "Recyclable Kraft Paper / Air Pillows",
        "filling_density_g_cm3": 0.015,
        "handling_instructions": "STANDARD GOODS — HANDLE WITH CARE",
        "orientation_recommendation": "Any direction",
        "weight_factor_multiplier": 0.8,
        "item_density_g_cm3": 0.22,
        "eco_rating": "A+",
        "co2_factor_g_per_cm3": 0.008,
        "badge_color": "#00b09b",
        "warning_flags": ["Standard Goods", "Recyclable Packaging"]
    },
    "Books/Stationery": {
        "cardboard_type": "Rigid Single-Wall / Heavy Kraft Mailer Box",
        "cardboard_gsm": 280,
        "padding_buffer_cm": 1.0,
        "filling_material": "Reinforced Edge/Corner Protectors + Kraft Void Fill",
        "filling_density_g_cm3": 0.045,
        "handling_instructions": "HEAVY PACKAGE — KEEP DRY & DO NOT BEND",
        "orientation_recommendation": "Flat Horizontal Stack",
        "weight_factor_multiplier": 1.8,
        "item_density_g_cm3": 0.75,
        "eco_rating": "A+",
        "co2_factor_g_per_cm3": 0.006,
        "badge_color": "#9b51e0",
        "warning_flags": ["Heavy Freight", "Do Not Bend", "Keep Dry"]
    },
    "Clothing/Textiles": {
        "cardboard_type": "Single-Wall Lightweight Corrugated / Kraft Box",
        "cardboard_gsm": 180,
        "padding_buffer_cm": 0.5,
        "filling_material": "Protective Polybag Seal (Minimal Void Fill Required)",
        "filling_density_g_cm3": 0.008,
        "handling_instructions": "SOFT GOODS — DO NOT USE BLADES FOR OPENING",
        "orientation_recommendation": "Any direction",
        "weight_factor_multiplier": 0.4,
        "item_density_g_cm3": 0.12,
        "eco_rating": "A+",
        "co2_factor_g_per_cm3": 0.004,
        "badge_color": "#f2994a",
        "warning_flags": ["Soft Goods", "No Blades"]
    },
    "Heavy Tools/Hardware": {
        "cardboard_type": "Triple-Wall Heavy Duty Corrugated (48 ECT)",
        "cardboard_gsm": 550,
        "padding_buffer_cm": 3.0,
        "filling_material": "Heavy Duty Foam Padding + Corner Strapping",
        "filling_density_g_cm3": 0.060,
        "handling_instructions": "HEAVY FREIGHT — USE TEAM LIFT — REINFORCED BOTTOM",
        "orientation_recommendation": "Flat Center Base",
        "weight_factor_multiplier": 2.2,
        "item_density_g_cm3": 1.85,
        "eco_rating": "A",
        "co2_factor_g_per_cm3": 0.025,
        "badge_color": "#eab308",
        "warning_flags": ["Heavy Freight", "Team Lift Required", "Reinforced Box"]
    },
    "Other/Miscellaneous": {
        "cardboard_type": "Standard Single-Wall Corrugated (32 ECT)",
        "cardboard_gsm": 230,
        "padding_buffer_cm": 2.0,
        "filling_material": "Standard Bubble Wrap + Air Pillows",
        "filling_density_g_cm3": 0.020,
        "handling_instructions": "STANDARD PACKAGING — HANDLE WITH CARE",
        "orientation_recommendation": "Any direction",
        "weight_factor_multiplier": 1.0,
        "item_density_g_cm3": 0.30,
        "eco_rating": "A",
        "co2_factor_g_per_cm3": 0.010,
        "badge_color": "#2f80ed",
        "warning_flags": ["Standard Goods"]
    }
}


def compute_comprehensive_packaging_metrics(
    category: str,
    obj_l: float,
    obj_w: float,
    obj_h: float,
    optimal_box: dict,
    stock_box: dict
) -> Dict[str, Any]:
    """
    Calculate estimated object weight, box surface area weight, void fill quantity,
    eco-sustainability rating, carbon footprint, and shipping estimates.
    """
    rules = get_packaging_rules(category)
    
    # 1. Object Volume & Estimated Weight
    obj_vol_cm3 = obj_l * obj_w * obj_h
    density = rules.get("item_density_g_cm3", 0.30)
    obj_weight_g = round(obj_vol_cm3 * density, 1)

    # 2. Box Weight (Surface Area calculation: 2*(L*W + L*H + W*H) in m2 * gsm)
    def calc_box_weight(box_dims_dict, gsm):
        if not box_dims_dict:
            return 0.0
        bl, bw, bh = box_dims_dict["length_cm"], box_dims_dict["width_cm"], box_dims_dict["height_cm"]
        surface_area_m2 = 2.0 * ((bl * bw) + (bl * bh) + (bw * bh)) / 10000.0
        return round(surface_area_m2 * gsm, 1)

    gsm = rules.get("cardboard_gsm", 250)
    opt_box_weight_g = calc_box_weight(optimal_box, gsm)
    stock_box_weight_g = calc_box_weight(stock_box, gsm) if stock_box else opt_box_weight_g

    # 3. Filling Material Volume & Weight
    target_box_vol = stock_box["volume_cm3"] if stock_box else optimal_box["volume_cm3"]
    void_volume_cm3 = max(0.0, target_box_vol - obj_vol_cm3)
    fill_density = rules.get("filling_density_g_cm3", 0.020)
    filling_weight_g = round(void_volume_cm3 * fill_density, 1)

    # 4. Total Package Weight
    active_box_weight = stock_box_weight_g if stock_box else opt_box_weight_g
    total_package_weight_g = round(obj_weight_g + active_box_weight + filling_weight_g, 1)

    # 5. Sustainability & Carbon Metrics
    co2_factor = rules.get("co2_factor_g_per_cm3", 0.010)
    estimated_co2_g = round(target_box_vol * co2_factor, 1)
    
    # Shipping Tier Estimate
    weight_kg = total_package_weight_g / 1000.0
    if weight_kg < 0.5:
        ship_tier = "Standard Light Parcel ($3.50 - $5.00)"
    elif weight_kg < 2.0:
        ship_tier = "Medium Express Freight ($7.50 - $12.00)"
    else:
        ship_tier = "Heavy Cargo Tier ($15.00+)"

    return {
        "rules": rules,
        "object_metrics": {
            "volume_cm3": round(obj_vol_cm3, 1),
            "estimated_weight_g": obj_weight_g,
            "estimated_weight_kg": round(obj_weight_g / 1000.0, 3)
        },
        "box_metrics": {
            "cardboard_type": rules["cardboard_type"],
            "optimal_box_weight_g": opt_box_weight_g,
            "stock_box_weight_g": stock_box_weight_g,
        },
        "filling_metrics": {
            "material_name": rules["filling_material"],
            "void_volume_cm3": round(void_volume_cm3, 1),
            "material_weight_g": filling_weight_g
        },
        "total_package": {
            "total_weight_g": total_package_weight_g,
            "total_weight_kg": round(total_package_weight_g / 1000.0, 3)
        },
        "environmental_and_shipping": {
            "eco_rating": rules.get("eco_rating", "A"),
            "estimated_co2_g": estimated_co2_g,
            "shipping_tier": ship_tier,
            "warning_flags": rules.get("warning_flags", ["Standard Goods"])
        }
    }


def get_packaging_rules(category: str) -> Dict[str, Any]:
    """
    Retrieve packaging rules for a given category.
    Falls back to 'Other/Miscellaneous' if category is unmapped.
    """
    category_clean = category.strip()
    if category_clean in RULES_TABLE:
        return RULES_TABLE[category_clean]
    
    # Partial match fallback logic
    for cat_name, rules in RULES_TABLE.items():
        if cat_name.lower() in category_clean.lower() or category_clean.lower() in cat_name.lower():
            return rules
            
    return RULES_TABLE["Other/Miscellaneous"]

