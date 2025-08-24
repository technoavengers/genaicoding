// Represents a product in Walmart's ecommerce system
public class Product {
    private String product_id;
    private String product_name;
    private String product_description;
    private double product_price;
    private String product_category; // Added category field

    // Constructor
    public Product(String product_id, String product_name, String product_description, double product_price, String product_category) {
        this.product_id = product_id;
        this.product_name = product_name;
        this.product_description = product_description;
        this.product_price = product_price;
        this.product_category = product_category;
    }

    // Getters and Setters
    public String getId() {
        return product_id;
    }

    public void setId(String product_id) {
        this.product_id = product_id;
    }

    public String getName() {
        return product_name;
    }

    public void setName(String product_name) {
        this.product_name = product_name;
    }

    public String getDescription() {
        return product_description;
    }

    public void setDescription(String product_description) {
        this.product_description = product_description;
    }

    public double getPrice() {
        return product_price;
    }

    public void setPrice(double product_price) {
        this.product_price = product_price;
    }

    public String getCategory() {
        return product_category;
    }

    public void setCategory(String product_category) {
        this.product_category = product_category;
    }
}
