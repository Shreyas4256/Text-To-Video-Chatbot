import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { nextSecureHeaders } from 'next-secure-headers';

export function middleware(request: NextRequest) {
  const response = NextResponse.next();
  nextSecureHeaders()(request, response);
  return response;
}
