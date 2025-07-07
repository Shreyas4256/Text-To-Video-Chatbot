interface Params { params: { slug: string } }

export default function BlogPostPage({ params }: Params) {
  return <h1>Blog {params.slug}</h1>;
}
