"use client";

import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const schema = z.object({
  name: z.string().min(1),
  message: z.string().min(1),
});

type FormData = z.infer<typeof schema>;

export default function ContactForm() {
  const { register, handleSubmit } = useForm<FormData>({ resolver: zodResolver(schema) });

  const onSubmit = handleSubmit((data) => {
    console.log(data);
  });

  return (
    <form onSubmit={onSubmit}>
      <input {...register('name')} placeholder="Name" />
      <textarea {...register('message')} placeholder="Message" />
      <button type="submit">Send</button>
    </form>
  );
}
