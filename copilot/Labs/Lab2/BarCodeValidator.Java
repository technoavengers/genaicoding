public class BarcodeValidator {
    public static boolean isValidBarcode(String barcode) {
        return barcode.matches("\\d{12}");
    }
}

