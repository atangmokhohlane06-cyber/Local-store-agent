import pandas as pd
from langchain.tools import tool

CSV_PATH = "products.csv"


def _df() -> pd.DataFrame:
    # This reads the CSV fresh on every single call so any edits you save are instantly live!
    return pd.read_csv(CSV_PATH)


@tool
def search_products(query: str) -> str:
    """Search products by name, category, or description keywords."""
    df = _df()
    q = query.lower()

    # Check if the search query matches name, category, or description columns
    hits = df[
        df["name"].str.lower().str.contains(q)
        | df["category"].str.lower().str.contains(q)
        | df["description"].str.lower().str.contains(q)
    ].head(5)

    if hits.empty:
        return "No matching products found in our catalog."

    # Return basic information back to the LLM as a JSON string
    return hits[["id", "name", "price", "stock"]].to_json(orient="records")


@tool
def get_product(product_id:str) -> str:
    """Get detailed information about a product using its ID code (e.g., MIL-HON-500)."""
    df = _df()
    row = df[df["id"].str.upper() == product_id.upper()]

    if row.empty:
        return f"Could not find any product matching ID code {product_id}."

    # Return all details back to the LLM as a JSON string
    return row.iloc[0].to_json()

@tool
def check_stock(product_id: str) -> str:
    """Check the exact stock availability of a product using its ID code (e.g., MIL-HON-500)."""
    df = _df()
    row = df[df["id"].str.upper() == product_id.upper()]

    if row.empty:
        return f"Could not find any product matching ID code {product_id}."

    return (
        f"{row.iloc[0]['name']}: {int(row.iloc[0]['stock'])} units currently in stock."
    )

@tool 
def recommend(category:str) -> str:
    """Recommend up to 3 in-stock products in a given category."""
    df=_df()
    c = category.lower()
    
    hits=df[(df["category"].str.lower() == category.lower())& (df["stock"]>0)].sort_values("stock", ascending=False).head(3)

    if hits.empty:
        return f"Sorry,Nothing available in '{category}'category."
        return hits[["id","name","price","stock"]].to_json(orient="records")   
    
    
    
# Export our tools list so the agent script can import them cleanly
TOOLS = [search_products, check_stock]
