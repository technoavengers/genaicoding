import React from 'react';

function Checkout({ cart }) {
    let total = 0;
    for (let i = 0; i < cart.length; i++) {
        total += cart[i].price;
    }
    return <h2>Total: ${total}</h2>;
}

export default Checkout;
