import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { createHeadersObject } from 'next-secure-headers';

export function middleware(request: NextRequest) {
  const response = NextResponse.next();
  const headers = createHeadersObject();
  for (const [key, value] of Object.entries(headers)) {
    response.headers.set(key, value);
  }
  return response;
}
