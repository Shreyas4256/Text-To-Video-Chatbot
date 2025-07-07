import Image from 'next/image';

export default function ProductCard() {
  return (
    <div>
      <Image src="/images/placeholder.png" alt="Product" width={200} height={200} />
      <h2>Product</h2>
    </div>
  );
}
