public class DiscountCalculator {

public static double calculateFinalPrice(double price, String category) {
    double discount = 0.0;
    if (category.equals("electronics")) {
        discount = 0.1; // 10% discount
    } else if (category.equals("clothing")) {
        discount = 0.2; // 20% discount
    }
    double finalPrice = price * discount;
    return finalPrice;
}   

public static void main(String[] args) {
    double finalPrice = calculateFinalPrice(100.0, "electronics");
    System.out.println("Final Price: " + finalPrice);
}


}