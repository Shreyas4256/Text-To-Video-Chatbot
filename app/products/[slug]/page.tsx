interface Params { params: { slug: string } }

export default function ProductSlugPage({ params }: Params) {
  return <h1>Product {params.slug}</h1>;
}
