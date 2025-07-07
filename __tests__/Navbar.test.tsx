import { render, screen } from '@testing-library/react';
import Navbar from '../components/Navbar';

test('renders navbar', () => {
  render(<Navbar />);
  expect(screen.getByText(/navbar/i)).toBeInTheDocument();
});
